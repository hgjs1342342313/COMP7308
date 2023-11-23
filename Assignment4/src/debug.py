def rrt_near(self, x_target):
		r"""
		Question 2(a)
		Find nodes within the rrt/rrt_star tree 
		whose distance to the target node is within radius r.

		Parameters that you may use:
		@param self.nodes: a rrt/rrt_star tree, a list containing all the nodes
					 Access by (example):
					 node = self.nodes[2]
					 node is a rrt/rrt_star node, which is a list containing:
					 for 'rrt': [state, parent_node_index]
					 for 'rrt_star': [state, parent_node_index, path_cost]
					 Access by (example):
					 node[0]: 2*1 matrix, state
					 node[1]: int, parent_node_index, usage as self.nodes[parent_node_index]
					 node[2]: float, path_cost, the path length from root node to this node
		@param x_target: the state of the target node
		@param eta: a constant described within the algorithm
		@param gamma: a constant described within the algorithm
		@param d: dimension of the world

		Target:
		@param node_near_list: a list containing nodes (NOT states of the nodes)
							   and their index  within the rrt/rrt_star tree. 
							   These nodes' distances to x_target are within radius r,
							   where r is described within the algorithm.
							   Form: [[node1, index1], [node2, index2]]
		"""
		gamma = 5
		eta = 0.6*gamma
		d = 2
		node_near_list = []
		"*** YOUR CODE STARTS HERE ***"
		# Calculate r 
		card = len(self.nodes)
		foo = math.log(card)/card
		foo = foo ** (1/d)
		foo = foo*gamma
		r = min(foo, eta)
		x = x_target[0,0]
		y = x_target[1, 0]
		# Loop within tree to find all satisfiable node
		for i, node in enumerate(self.nodes):
			x_new = node[0][0,0]
			y_new = node[0][1,0]
			distanc = math.sqrt((x-x_new)**2+(y-y_new)**2)
			if distanc < r:
				satisfied_node = [node, i]
				node_near_list.append(satisfied_node)
		"*** YOUR CODE ENDS HERE ***"

		return node_near_list

def rrt(self):
		r"""
		Question 1(c)
		Finish the part of rrt algorithm.
		AND
		Question 2(b)
		Finish the part of rrt_star algorithm.

		Parameters that you may use:
		@param self.nodes: a rrt/rrt_star tree, a list containing all the nodes
					 Access by (example):
					 node = self.nodes[2]
					 node is a rrt/rrt_star node, which is a list containing:
					 for 'rrt': [state, parent_node_index]
					 for 'rrt_star': [state, parent_node_index, path_cost]
					 Access by (example):
					 node[0]: 2*1 matrix, state
					 node[1]: int, parent_node_index, usage as self.nodes[parent_node_index]
					 node[2]: float, path_cost, the path length from root node to this node
		@param x_nearest, index_nearest
		@param x_new

		Function:
		@func self.rrt_near
		@func self.rrt_collision_free
		@func self.rrt_cost

		Target:
		Finish the part of rrt algorithm.
		AND
		Finish the part of rrt_star algorithm.
		"""
		for i in range(self.p_step_per_iter):
			x_rand = self.rrt_sample_free()
			x_nearest, index_nearest = self.rrt_nearest(x_rand)
			x_new = self.rrt_stear(x_nearest, x_rand)
			if self.rrt_collision_free(x_nearest, x_new):
				if self.p_mode == 'rrt':
					"*** YOUR CODE STARTS HERE ***"
					# Here Question 1(c) starts
					# Line 7, save the new node to our list
					new_node = [x_new, index_nearest]
					self.nodes.append(new_node)
					"*** YOUR CODE ENDS HERE ***"
				elif self.p_mode == 'rrt_star':
					"*** YOUR CODE STARTS HERE ***"
					# Here Question 2(b) starts
					# Line 7, find nodes near new node with state x_new,
					# get a list saving these nodes and their indexes
					Xnear = self.rrt_near(x_new)
					# Line 9-12, loop from the list, find node with
					# the minimum path cost, get its index as well
					xmin = x_nearest
					imin = index_nearest
					cmin = self.nodes[index_nearest][2]+self.rrt_cost(x_nearest, x_new)
					# lin9 上面没问题
					for item in Xnear:
						x_node = item[0] # 一个node
						x_ind = item[1] # node的自己的index
						# glob_x_ind = x_node[1] parent的index
						x_near = x_node[0] # node的state
						if self.rrt_collision_free(x_near, x_new) and self.nodes[x_ind][2] + self.rrt_cost(x_near, x_new) < cmin:
							xmin = x_near
							imin = x_ind
							cmin = self.nodes[x_ind][2] + self.rrt_cost(x_near, x_new)

					# Line 8 and 13, save the new node to our rrt_star tree
					# with form [state, parent_node_index, path_cost]
					new_node = [x_new, imin, cmin]
					self.nodes.append(new_node)
					new_node_index = len(self.nodes)-1

					# Line 14-16, loop from the list, check whether these nodes
					# can get a shorter path cost when they connect to our new node
					# If so, update their connection and the cost.
					for item in Xnear:
						x_node = item[0] # 一个node
						x_ind = item[1] # node的自己的index
						# glob_x_ind = x_node[1] parent的index
						x_near = x_node[0] # node的state
						if self.rrt_collision_free(x_new, x_near) and cmin + self.rrt_cost(x_new, x_near) < self.nodes[x_ind][2]:
							x_node[1] = new_node_index

					"*** YOUR CODE ENDS HERE ***"

			if self.rrt_reach_goal(x_new):
				self.p_status = 'successful'
				print(f'{self.p_mode} succeeds in finding a policy.') 
				self.rrt_generate_results()
				return

		self.p_current_iter += 1
		if self.p_current_iter >= self.p_max_iter:
			self.p_status == 'failed'
			print(f'{self.p_mode} fails to find a policy.') 
		return
